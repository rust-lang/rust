#!/usr/bin/env python2.7

"""
Testing dec2flt
===============
These are *really* extensive tests. Expect them to run for hours. Due to the
nature of the problem (the input is a string of arbitrary length), exhaustive
testing is not really possible. Instead, there are exhaustive tests for some
classes of inputs for which that is feasible and a bunch of deterministic and
random non-exhaustive tests for covering everything else.

The actual tests (generating decimal strings and feeding them to dec2flt) is
performed by a set of stand-along rust programs. This script compiles, runs,
and supervises them. The programs report the strings they generate and the
floating point numbers they converted those strings to, and this script
checks that the results are correct.

You can run specific tests rather than all of them by giving their names
(without .rs extension) as command line parameters.

Verification
------------
The tricky part is not generating those inputs but verifying the outputs.
Comparing with the result of Python's float() does not cut it because
(and this is apparently undocumented) although Python includes a version of
Martin Gay's code including the decimal-to-float part, it doesn't actually use
it for float() (only for round()) instead relying on the system scanf() which
is not necessarily completely accurate.

Instead, we take the input and compute the true value with bignum arithmetic
(as a fraction, using the ``fractions`` module).

Given an input string and the corresponding float computed via Rust, simply
decode the float into f * 2^k (for integers f, k) and the ULP.
We can now easily compute the error and check if it is within 0.5 ULP as it
should be. Zero and infinites are handled similarly:

- If the approximation is 0.0, the exact value should be *less or equal*
  half the smallest denormal float: the smallest denormal floating point
  number has an odd mantissa (00...001) and thus half of that is rounded
  to 00...00, i.e., zero.
- If the approximation is Inf, the exact value should be *greater or equal*
  to the largest finite float + 0.5 ULP: the largest finite float has an odd
  mantissa (11...11), so that plus half an ULP is rounded up to the nearest
  even number, which overflows.

Implementation details
----------------------
This directory contains a set of single-file Rust programs that perform
tests with a particular class of inputs. Each is compiled and run without
parameters, outputs (f64, f32, decimal) pairs to verify externally, and
in any case either exits gracefully or with a panic.

If a test binary writes *anything at all* to stderr or exits with an
exit code that's not 0, the test fails.
The output on stdout is treated as (f64, f32, decimal) record, encoded thusly:

- First, the bits of the f64 encoded as an ASCII hex string.
- Second, the bits of the f32 encoded as an ASCII hex string.
- Then the corresponding string input, in ASCII
- The record is terminated with a newline.

Incomplete records are an error. Not-a-Number bit patterns are invalid too.

The tests run serially but the validation for a single test is parallelized
with ``multiprocessing``. Each test is launched as a subprocess.
One thread supervises it: Accepts and enqueues records to validate, observe
stderr, and waits for the process to exit. A set of worker processes perform
the validation work for the outputs enqueued there. Another thread listens
for progress updates from the workers.

Known issues
------------
Some errors (e.g., NaN outputs) aren't handled very gracefully.
Also, if there is an exception or the process is interrupted (at least on
Windows) the worker processes are leaked and stick around forever.
They're only a few megabytes each, but still, this script should not be run
if you aren't prepared to manually kill a lot of orphaned processes.
"""
from __future__ import print_function
import sys
import os.path
import time
import struct
from fractions import Fraction
from collections import namedtuple
from subprocess import Popen, check_call, PIPE
from glob import glob
import multiprocessing
import threading
import ctypes
import binascii

try:  # Python 3
    import queue as Queue
except ImportError:  # Python 2
    import Queue

NUM_WORKERS = 2
UPDATE_EVERY_N = 50000
INF = namedtuple('INF', '')()
NEG_INF = namedtuple('NEG_INF', '')()
ZERO = namedtuple('ZERO', '')()
MAILBOX = None  # The queue for reporting errors to the main process.
STDOUT_LOCK = threading.Lock()
test_name = None
child_processes = []
exit_status = 0

def msg(*args):
    with STDOUT_LOCK:
        print("[" + test_name + "]", *args)
        sys.stdout.flush()


def write_errors():
    global exit_status
    f = open("errors.txt", 'w')
    have_seen_error = False
    while True:
        args = MAILBOX.get()
        if args is None:
            f.close()
            break
        print(*args, file=f)
        f.flush()
        if not have_seen_error:
            have_seen_error = True
            msg("Something is broken:", *args)
            msg("Future errors logged to errors.txt")
            exit_status = 101


def rustc(test):
    rs = test + '.rs'
    exe = test + '.exe'  # hopefully this makes it work on *nix
    print("compiling", test)
    sys.stdout.flush()
    check_call(['rustc', rs, '-o', exe])


def run(test):
    global test_name
    test_name = test

    t0 = time.clock()
    msg("setting up supervisor")
    exe = test + '.exe'
    proc = Popen(exe, bufsize=1<<20 , stdin=PIPE, stdout=PIPE, stderr=PIPE)
    done = multiprocessing.Value(ctypes.c_bool)
    queue = multiprocessing.Queue(maxsize=5)#(maxsize=1024)
    workers = []
    for n in range(NUM_WORKERS):
        worker = multiprocessing.Process(name='Worker-' + str(n + 1),
                                         target=init_worker,
                                         args=[test, MAILBOX, queue, done])
        workers.append(worker)
        child_processes.append(worker)
    for worker in workers:
        worker.start()
    msg("running test")
    interact(proc, queue)
    with done.get_lock():
        done.value = True
    for worker in workers:
        worker.join()
    msg("python is done")
    assert queue.empty(), "did not validate everything"
    dt = time.clock() - t0
    msg("took", round(dt, 3), "seconds")


def interact(proc, queue):
    n = 0
    while proc.poll() is None:
        line = proc.stdout.readline()
        if not line:
            continue
        assert line.endswith('\n'), "incomplete line: " + repr(line)
        queue.put(line)
        n += 1
        if n % UPDATE_EVERY_N == 0:
            msg("got", str(n // 1000) + "k", "records")
    msg("rust is done. exit code:", proc.returncode)
    rest, stderr = proc.communicate()
    if stderr:
        msg("rust stderr output:", stderr)
    for line in rest.split('\n'):
        if not line:
            continue
        queue.put(line)


def main():
    global MAILBOX
    tests = [os.path.splitext(f)[0] for f in glob('*.rs')
                                    if not f.startswith('_')]
    whitelist = sys.argv[1:]
    if whitelist:
        tests = [test for test in tests if test in whitelist]
    if not tests:
        print("Error: No tests to run")
        sys.exit(1)
    # Compile first for quicker feedback
    for test in tests:
        rustc(test)
    # Set up mailbox once for all tests
    MAILBOX = multiprocessing.Queue()
    mailman = threading.Thread(target=write_errors)
    mailman.daemon = True
    mailman.start()
    for test in tests:
        if whitelist and test not in whitelist:
            continue
        run(test)
    MAILBOX.put(None)
    mailman.join()


# ---- Worker thread code ----


POW2 = { e: Fraction(2) ** e for e in range(-1100, 1100) }
HALF_ULP = { e: (Fraction(2) ** e)/2 for e in range(-1100, 1100) }
DONE_FLAG = None


def send_error_to_supervisor(*args):
    MAILBOX.put(args)


def init_worker(test, mailbox, queue, done):
    global test_name, MAILBOX, DONE_FLAG
    test_name = test
    MAILBOX = mailbox
    DONE_FLAG = done
    do_work(queue)


def is_done():
    with DONE_FLAG.get_lock():
        return DONE_FLAG.value


def do_work(queue):
    while True:
        try:
            line = queue.get(timeout=0.01)
        except Queue.Empty:
            if queue.empty() and is_done():
                return
            else:
                continue
        bin64, bin32, text = line.rstrip().split()
        validate(bin64, bin32, text)


def decode_binary64(x):
    """
    Turn a IEEE 754 binary64 into (mantissa, exponent), except 0.0 and
    infinity (positive and negative), which return ZERO, INF, and NEG_INF
    respectively.
    """
    x = binascii.unhexlify(x)
    assert len(x) == 8, repr(x)
    [bits] = struct.unpack(b'>Q', x)
    if bits == 0:
        return ZERO
    exponent = (bits >> 52) & 0x7FF
    negative = bits >> 63
    low_bits = bits & 0xFFFFFFFFFFFFF
    if exponent == 0:
        mantissa = low_bits
        exponent += 1
        if mantissa == 0:
            return ZERO
    elif exponent == 0x7FF:
        assert low_bits == 0, "NaN"
        if negative:
            return NEG_INF
        else:
            return INF
    else:
        mantissa = low_bits | (1 << 52)
    exponent -= 1023 + 52
    if negative:
        mantissa = -mantissa
    return (mantissa, exponent)


def decode_binary32(x):
    """
    Turn a IEEE 754 binary32 into (mantissa, exponent), except 0.0 and
    infinity (positive and negative), which return ZERO, INF, and NEG_INF
    respectively.
    """
    x = binascii.unhexlify(x)
    assert len(x) == 4, repr(x)
    [bits] = struct.unpack(b'>I', x)
    if bits == 0:
        return ZERO
    exponent = (bits >> 23) & 0xFF
    negative = bits >> 31
    low_bits = bits & 0x7FFFFF
    if exponent == 0:
        mantissa = low_bits
        exponent += 1
        if mantissa == 0:
            return ZERO
    elif exponent == 0xFF:
        if negative:
            return NEG_INF
        else:
            return INF
    else:
        mantissa = low_bits | (1 << 23)
    exponent -= 127 + 23
    if negative:
        mantissa = -mantissa
    return (mantissa, exponent)


MIN_SUBNORMAL_DOUBLE = Fraction(2) ** -1074
MIN_SUBNORMAL_SINGLE = Fraction(2) ** -149  # XXX unsure
MAX_DOUBLE = (2 - Fraction(2) ** -52) * (2 ** 1023)
MAX_SINGLE = (2 - Fraction(2) ** -23) * (2 ** 127)
MAX_ULP_DOUBLE = 1023 - 52
MAX_ULP_SINGLE = 127 - 23
DOUBLE_ZERO_CUTOFF = MIN_SUBNORMAL_DOUBLE / 2
DOUBLE_INF_CUTOFF = MAX_DOUBLE + 2 ** (MAX_ULP_DOUBLE - 1)
SINGLE_ZERO_CUTOFF = MIN_SUBNORMAL_SINGLE / 2
SINGLE_INF_CUTOFF = MAX_SINGLE + 2 ** (MAX_ULP_SINGLE - 1)

def validate(bin64, bin32, text):
    double = decode_binary64(bin64)
    single = decode_binary32(bin32)
    real = Fraction(text)

    if double is ZERO:
        if real > DOUBLE_ZERO_CUTOFF:
            record_special_error(text, "f64 zero")
    elif double is INF:
        if real < DOUBLE_INF_CUTOFF:
            record_special_error(text, "f64 inf")
    elif double is NEG_INF:
        if -real < DOUBLE_INF_CUTOFF:
            record_special_error(text, "f64 -inf")
    elif len(double) == 2:
        sig, k = double
        validate_normal(text, real, sig, k, "f64")
    else:
        assert 0, "didn't handle binary64"
    if single is ZERO:
        if real > SINGLE_ZERO_CUTOFF:
            record_special_error(text, "f32 zero")
    elif single is INF:
        if real < SINGLE_INF_CUTOFF:
            record_special_error(text, "f32 inf")
    elif single is NEG_INF:
        if -real < SINGLE_INF_CUTOFF:
            record_special_error(text, "f32 -inf")
    elif len(single) == 2:
        sig, k = single
        validate_normal(text, real, sig, k, "f32")
    else:
        assert 0, "didn't handle binary32"

def record_special_error(text, descr):
    send_error_to_supervisor(text.strip(), "wrongly rounded to", descr)


def validate_normal(text, real, sig, k, kind):
    approx = sig * POW2[k]
    error = abs(approx - real)
    if error > HALF_ULP[k]:
        record_normal_error(text, error, k, kind)


def record_normal_error(text, error, k, kind):
    one_ulp = HALF_ULP[k + 1]
    assert one_ulp == 2 * HALF_ULP[k]
    relative_error = error / one_ulp
    text = text.strip()
    try:
        err_repr = float(relative_error)
    except ValueError:
        err_repr = str(err_repr).replace('/', ' / ')
    send_error_to_supervisor(err_repr, "ULP error on", text, "(" + kind + ")")


if __name__ == '__main__':
    main()
