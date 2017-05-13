The goal of the libc crate is to have CI running everywhere to have the
strongest guarantees about the definitions that this library contains, and as a
result the CI is pretty complicated and also pretty large! Hopefully this can
serve as a guide through the sea of scripts in this directory and elsewhere in
this project.

# Files

First up, let's talk about the files in this directory:

* `run-travis.sh` - a shell script run by all Travis builders, this is
  responsible for setting up the rest of the environment such as installing new
  packages, downloading Rust target libraries, etc.

* `run.sh` - the actual script which runs tests for a particular architecture.
  Called from the `run-travis.sh` script this will run all tests for the target
  specified.

* `cargo-config` - Cargo configuration of linkers to use copied into place by
  the `run-travis.sh` script before builds are run.

* `dox.sh` - script called from `run-travis.sh` on only the linux 64-bit nightly
  Travis bots to build documentation for this crate.

* `landing-page-*.html` - used by `dox.sh` to generate a landing page for all
  architectures' documentation.

* `run-qemu.sh` - see discussion about QEMU below

* `mips`, `rumprun` - instructions to build the docker image for each respective
  CI target

# CI Systems

Currently this repository leverages a combination of Travis CI and AppVeyor for
running tests. The triples tested are:

* AppVeyor
  * `{i686,x86_64}-pc-windows-{msvc,gnu}`
* Travis
  * `{i686,x86_64,mips,aarch64}-unknown-linux-gnu`
  * `x86_64-unknown-linux-musl`
  * `arm-unknown-linux-gnueabihf`
  * `arm-linux-androideabi`
  * `{i686,x86_64}-apple-{darwin,ios}`
  * `x86_64-rumprun-netbsd`
  * `x86_64-unknown-freebsd`
  * `x86_64-unknown-openbsd`

The Windows triples are all pretty standard, they just set up their environment
then run tests, no need for downloading any extra target libs (we just download
the right installer). The Intel Linux/OSX builds are similar in that we just
download the right target libs and run tests. Note that the Intel Linux/OSX
builds are run on stable/beta/nightly, but are the only ones that do so.

The remaining architectures look like:

* Android runs in a [docker image][android-docker] with an emulator, the NDK,
  and the SDK already set up. The entire build happens within the docker image.
* The MIPS, ARM, and AArch64 builds all use the QEMU userspace emulator to run
  the generated binary to actually verify the tests pass.
* The MUSL build just has to download a MUSL compiler and target libraries and
  then otherwise runs tests normally.
* iOS builds need an extra linker flag currently, but beyond that they're built
  as standard as everything else.
* The rumprun target builds an entire kernel from the test suite and then runs
  it inside QEMU using the serial console to test whether it succeeded or
  failed.
* The BSD builds, currently OpenBSD and FreeBSD, use QEMU to boot up a system
  and compile/run tests. More information on that below.

[android-docker]: https://github.com/rust-lang/rust-buildbot/blob/master/slaves/android/Dockerfile

## QEMU

Lots of the architectures tested here use QEMU in the tests, so it's worth going
over all the crazy capabilities QEMU has and the various flavors in which we use
it!

First up, QEMU has userspace emulation where it doesn't boot a full kernel, it
just runs a binary from another architecture (using the `qemu-<arch>` wrappers).
We provide it the runtime path for the dynamically loaded system libraries,
however. This strategy is used for all Linux architectures that aren't intel.
Note that one downside of this QEMU system is that threads are barely
implemented, so we're careful to not spawn many threads.

For the rumprun target the only output is a kernel image, so we just use that
plus the `rumpbake` command to create a full kernel image which is then run from
within QEMU.

Finally, the fun part, the BSDs. Quite a few hoops are jumped through to get CI
working for these platforms, but the gist of it looks like:

* Cross compiling from Linux to any of the BSDs seems to be quite non-standard.
  We may be able to get it working but it might be difficult at that point to
  ensure that the libc definitions align with what you'd get on the BSD itself.
  As a result, we try to do compiles within the BSD distro.
* On Travis we can't run a VM-in-a-VM, so we resort to userspace emulation
  (QEMU).
* Unfortunately on Travis we also can't use KVM, so the emulation is super slow.

With all that in mind, the way BSD is tested looks like:

1. Download a pre-prepared image for the OS being tested.
2. Generate the tests for the OS being tested. This involves running the `ctest`
   library over libc to generate a Rust file and a C file which will then be
   compiled into the final test.
3. Generate a disk image which will later be mounted by the OS being tested.
   This image is mostly just the libc directory, but some modifications are made
   to compile the generated files from step 2.
4. The kernel is booted in QEMU, and it is configured to detect the libc-test
   image being available, run the test script, and then shut down afterwards.
5. Look for whether the tests passed in the serial console output of the kernel.

There's some pretty specific instructions for setting up each image (detailed
below), but the main gist of this is that we must avoid a vanilla `cargo run`
inside of the `libc-test` directory (which is what it's intended for) because
that would compile `syntex_syntax`, a large library, with userspace emulation.
This invariably times out on Travis, so we can't do that.

Once all those hoops are jumped through, however, we can be happy that we're
testing almost everything!

Below are some details of how to set up the initial OS images which are
downloaded. Each image must be enabled have input/output over the serial
console, log in automatically at the serial console, detect if a second drive in
QEMU is available, and if so mount it, run a script (it'll specifically be
`run-qemu.sh` in this folder which is copied into the generated image talked
about above), and then shut down.

### QEMU setup - FreeBSD

1. Download CD installer (most minimal is fine)
2. `qemu-img create -f qcow2 foo.qcow2 2G`
3. `qemu -cdrom foo.iso -drive if=virtio,file=foo.qcow2 -net nic,model=virtio -net user`
4. run installer
5. `echo 'console="comconsole"' >> /boot/loader.conf`
6. `echo 'autoboot_delay="0"' >> /boot/loader.conf`
7. look at /etc/ttys, see what getty argument is for ttyu0
8. edit /etc/gettytab, look for ttyu0 argument, prepend `:al=root` to line
   beneath

(note that the current image has a `freebsd` user, but this isn't really
necessary)

Once that's done, arrange for this script to run at login:

```
#!/bin/sh

sudo kldload ext2fs
[ -e /dev/vtbd1 ] || exit 0
sudo mount -t ext2fs /dev/vtbd1 /mnt
sh /mnt/run.sh /mnt
sudo poweroff
```

Helpful links

* https://en.wikibooks.org/wiki/QEMU/Images
* https://blog.nekoconeko.nl/blog/2015/06/04/creating-an-openstack-freebsd-image.html
* https://www.freebsd.org/doc/handbook/serialconsole-setup.html


### QEMU setup - OpenBSD

1. Download CD installer
2. `qemu-img create -f qcow2 foo.qcow2 2G`
3. `qemu -cdrom foo.iso -drive if=virtio,file=foo.qcow2 -net nic,model=virtio -net user`
4. run installer
5. `echo 'set tty com0' >> /etc/boot.conf`
6. `echo 'boot' >> /etc/boot.conf`
7. Modify /etc/ttys, change the `tty00` at the end from 'unknown off' to
   'vt220 on secure'
8. Modify same line in /etc/ttys to have `"/root/foo.sh"` as the shell
9. Add this script to `/root/foo.sh`

```
#!/bin/sh
exec 1>/dev/tty00
exec 2>&1

if mount -t ext2fs /dev/sd1c /mnt; then
  sh /mnt/run.sh /mnt
  shutdown -ph now
fi

# limited shell...
exec /bin/sh < /dev/tty00
```

10. `chmod +x /root/foo.sh`

Helpful links:

* https://en.wikibooks.org/wiki/QEMU/Images
* http://www.openbsd.org/faq/faq7.html#SerCon

# Questions?

Hopefully that's at least somewhat of an introduction to everything going on
here, and feel free to ping @alexcrichton with questions!

