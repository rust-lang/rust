#!/usr/bin/env python3
# ignore-tidy-linelength

# This is a small script that we use on CI to collect CPU usage statistics of
# our builders. By seeing graphs of CPU usage over time we hope to correlate
# that with possible improvements to Rust's own build system, ideally diagnosing
# that either builders are always fully using their CPU resources or they're
# idle for long stretches of time.
#
# This script is relatively simple, but it's platform specific. Each platform
# (OSX/Windows/Linux) has a different way of calculating the current state of
# CPU at a point in time. We then compare two captured states to determine the
# percentage of time spent in one state versus another. The state capturing is
# all platform-specific but the loop at the bottom is the cross platform part
# that executes everywhere.
#
# # Viewing statistics
#
# All builders will upload their CPU statistics as CSV files to our S3 buckets.
# These URLS look like:
#
#   https://$bucket.s3.amazonaws.com/rustc-builds/$commit/cpu-$builder.csv
#
# for example
#
#   https://rust-lang-ci2.s3.amazonaws.com/rustc-builds/68baada19cd5340f05f0db15a3e16d6671609bcc/cpu-x86_64-apple.csv
#
# Each CSV file has two columns. The first is the timestamp of the measurement
# and the second column is the % of idle cpu time in that time slice. Ideally
# the second column is always zero.
#
# Once you've downloaded a file there's various ways to plot it and visualize
# it. For command line usage you use the `src/etc/cpu-usage-over-time-plot.sh`
# script in this repository.

import datetime
import sys
import time

# Python 3.3 changed the value of `sys.platform` on Linux from "linux2" to just
# "linux". We check here with `.startswith` to keep compatibility with older
# Python versions (especially Python 2.7).
if sys.platform.startswith('linux'):
    class State:
        def __init__(self):
            with open('/proc/stat', 'r') as file:
                data = file.readline().split()
            if data[0] != 'cpu':
                raise Exception('did not start with "cpu"')
            self.user = int(data[1])
            self.nice = int(data[2])
            self.system = int(data[3])
            self.idle = int(data[4])
            self.iowait = int(data[5])
            self.irq = int(data[6])
            self.softirq = int(data[7])
            self.steal = int(data[8])
            self.guest = int(data[9])
            self.guest_nice = int(data[10])

        def idle_since(self, prev):
            user = self.user - prev.user
            nice = self.nice - prev.nice
            system = self.system - prev.system
            idle = self.idle - prev.idle
            iowait = self.iowait - prev.iowait
            irq = self.irq - prev.irq
            softirq = self.softirq - prev.softirq
            steal = self.steal - prev.steal
            guest = self.guest - prev.guest
            guest_nice = self.guest_nice - prev.guest_nice
            total = user + nice + system + idle + iowait + irq + softirq + steal + guest + guest_nice
            return float(idle) / float(total) * 100

elif sys.platform == 'win32':
    from ctypes.wintypes import DWORD
    from ctypes import Structure, windll, WinError, GetLastError, byref

    class FILETIME(Structure):
        _fields_ = [
            ("dwLowDateTime", DWORD),
            ("dwHighDateTime", DWORD),
        ]

    class State:
        def __init__(self):
            idle, kernel, user = FILETIME(), FILETIME(), FILETIME()

            success = windll.kernel32.GetSystemTimes(
                byref(idle),
                byref(kernel),
                byref(user),
            )

            assert success, WinError(GetLastError())[1]

            self.idle = (idle.dwHighDateTime << 32) | idle.dwLowDateTime
            self.kernel = (kernel.dwHighDateTime << 32) | kernel.dwLowDateTime
            self.user = (user.dwHighDateTime << 32) | user.dwLowDateTime

        def idle_since(self, prev):
            idle = self.idle - prev.idle
            user = self.user - prev.user
            kernel = self.kernel - prev.kernel
            return float(idle) / float(user + kernel) * 100

elif sys.platform == 'darwin':
    from ctypes import *
    libc = cdll.LoadLibrary('/usr/lib/libc.dylib')

    class host_cpu_load_info_data_t(Structure):
        _fields_ = [("cpu_ticks", c_uint * 4)]

    host_statistics = libc.host_statistics
    host_statistics.argtypes = [
        c_uint,
        c_int,
        POINTER(host_cpu_load_info_data_t),
        POINTER(c_int)
    ]
    host_statistics.restype = c_int

    CPU_STATE_USER = 0
    CPU_STATE_SYSTEM = 1
    CPU_STATE_IDLE = 2
    CPU_STATE_NICE = 3
    class State:
        def __init__(self):
            stats = host_cpu_load_info_data_t()
            count = c_int(4) # HOST_CPU_LOAD_INFO_COUNT
            err = libc.host_statistics(
                libc.mach_host_self(),
                c_int(3), # HOST_CPU_LOAD_INFO
                byref(stats),
                byref(count),
            )
            assert err == 0
            self.system = stats.cpu_ticks[CPU_STATE_SYSTEM]
            self.user = stats.cpu_ticks[CPU_STATE_USER]
            self.idle = stats.cpu_ticks[CPU_STATE_IDLE]
            self.nice = stats.cpu_ticks[CPU_STATE_NICE]

        def idle_since(self, prev):
            user = self.user - prev.user
            system = self.system - prev.system
            idle = self.idle - prev.idle
            nice = self.nice - prev.nice
            return float(idle) / float(user + system + idle + nice) * 100.0

else:
    print('unknown platform', sys.platform)
    sys.exit(1)

cur_state = State()
print("Time,Idle")
while True:
    time.sleep(1)
    next_state = State()
    now = datetime.datetime.utcnow().isoformat()
    idle = next_state.idle_since(cur_state)
    print("%s,%s" % (now, idle))
    sys.stdout.flush()
    cur_state = next_state
