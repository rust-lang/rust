- Feature Name: expand-open-options
- Start Date: 2015-08-04
- RFC PR:
- Rust Issue:

# Summary

Document and expand the open options.


# Motivation

The options that can be passed to the os when opening a file vary between
systems. And even if the options seem the same or similar, there may be
unexpected corner cases.

This RFC attempts to
- describe the different corner cases and behaviour of various operating
  systems.
- describe the intended behaviour and interaction of Rusts options.
- remedy cross-platform inconsistencies.
- suggest extra options to expose more platform-specific options.


# Detailed design

## Access modes

### Read-only
Open a file for read-only.


### Write-only
Open a file for write-only.

If a file already exist, the contents of that file get overwritten, but it is
not truncated. Example:
```
// contents of file before: "aaaaaaaa"
file.write(b"bbbb")
// contents of file after: "bbbbaaaa"
```


### Read-write
This is the simple combinations of read-only and write-only.


### Append-mode
Append-mode is similar to write-only, but all writes always happen at the end of
the file. This mode is especially useful if multiple processes or threads write
to a single file, like a log file. The operating system guarantees all writes
are atomic: no writes get mangled because another process writes at the same
time. No guarantees are made about the order writes end up in the file though.

Note: sadly append-mode is not atomic on NFS filesystems.

One maybe obvious note when using append-mode: make sure that all data that
belongs together, is written the the file in one operation. This can be done
by concatenating strings before passing them to `write()`, or using a buffered
writer (with a more than adequately sized buffer) and calling `flush()` when the
message is complete.

_Implementation detail_: On Windows opening a file in append-mode has one flag
_less_, the right to change existing data is removed. On Unix opening a file in
append-mode has one flag _extra_, that sets the status of the file descriptor to
append-mode. You could say that on Windows write is a superset of append, while
on Unix append is a superset of write. 

Because of this append is treated as a separate access mode in Rust, and if
`.append(true)` is specified than `.write()` is ignored.


### Read-append
Writing to the file works exactly the same as in append-mode.

Reading is more difficult, and may involve a lot of seeking. When the file is
opened, the position for reading may be set at the end of the file, so you
should first seek to the beginning. Also after every write the position is set
to the end of the file. So before writing you should save the current position,
and restore it after the write.
```
try!(file.seek(SeekFrom::Start(0)));
try!(file.read(&mut buffer));
let pos = try!(file.seek(SeekFrom::Current(0)));
try!(file.write(b"foo"));
try!(file.seek(SeekFrom::Start(pos)));
try!(file.read(&mut buffer));
```

### No access mode set
On Windows it is possible to open a file without setting an access mode. You can
do practically nothing with the file, but you can read
[metadata](https://msdn.microsoft.com/en-us/library/windows/desktop/aa363788%28v=vs.85%29.aspx)
such as the file size or timestamp.

On Unix it is traditionally not possible to open a file without specifying the
access mode, because of the way the access flags where defined: something like
`O_RDONLY = 0`, `O_WRONLY = 1` and `O_RDWR = 2`. When no flags are set, the
access mode is `0` and you fall back to opening the file read-only.

Linux since version 2.6.39 has functionality similar to Windows by opening the
file with `O_RDONLY | O_PATH`. Since version 3.6 you can call `fstat` on a file
descriptor opened this way.

For what it's worth
[GNU/Hurd](http://www.gnu.org/software/libc/manual/html_node/Access-Modes.html)
allows opening files without an access mode, because it defines `O_RDONLY = 1`
and `O_WRONLY = 2`. It allows all operations on the file that do not involve
reading or writing the data, like `chmod`.

On Unix systems that fall back to opening the file read-only, Rust will fail
opening the file with `E_INVALID`. Otherwise, if for example you are developing
on OS X but forget to set `.read(true)` when opening a file, it would work on
OS X but not on other systems.


### Windows-specific
`.access_mode(FILE_READ_DATA)`

On Windows you can detail whether you want to have read and/or write access to
the files data, attributes and/or extended attributes. Managing permissions in
such detail has proven itself too difficult, and generally not worth it.

In Rust, `.read(true)` gives you read access to the data, attributes and
extended attributes. Similarly, `.write(true)` gives write access to those
three, and the right to append data beyond the current end of the file.

But if you want fine-grained control, with `access_mode` you have it.

`.access_mode()` overrides the access mode set with Rusts cross-platform
options. Reasons to do so:
- it is not possible to un-set the flags set by Rusts options;
- otherwise the cross-platform options have to be wrapped with `#[cfg(unix)]`,
  instead of only having to wrap the Windows-specific option.

As a reference, this are the flags set by Rusts access modes:

bit| flag                  | read  | write | read-write | append | read-append |
--:|:----------------------|:-----:|:-----:|:----------:|:------:|:-----------:|
   | **generic rights**    |       |       |            |        |             |
31 | GENERIC_READ          |  set  |       |    set     |        |     set     |
30 | GENERIC_WRITE         |       |  set  |    set     |        |             |
29 | GENERIC_EXECUTE       |       |       |            |        |             |
28 | GENERIC_ALL           |       |       |            |        |             |
   | **specific rights**   |       |       |            |        |             |
 0 | FILE_READ_DATA        |implied|       |  implied   |        |   implied   |
 1 | FILE_WRITE_DATA       |       |implied|  implied   |        |             |
 2 | FILE_APPEND_DATA      |       |implied|  implied   |  set   |     set     |
 3 | FILE_READ_EA          |implied|       |  implied   |        |   implied   |
 4 | FILE_WRITE_EA         |       |implied|  implied   |  set   |     set     |
 6 | FILE_EXECUTE          |       |       |            |        |             |
 7 | FILE_READ_ATTRIBUTES  |implied|       |  implied   |        |   implied   |
 8 | FILE_WRITE_ATTRIBUTES |       |implied|  implied   |  set   |     set     |
   | **standard rights**   |       |       |            |        |             |
16 | DELETE                |       |       |            |        |             |
17 | READ_CONTROL          |implied|implied|  implied   |  set   | set+implied |
18 | WRITE_DAC             |       |       |            |        |             |
19 | WRITE_OWNER           |       |       |            |        |             |
20 | SYNCHRONIZE           |implied|implied|  implied   |  set   | set+implied |

The implied flags can be specified explicitly with the constants
`FILE_GENERIC_READ` and `FILE_GENERIC_WRITE`.


## Creation modes

creation mode                | file exists | file does not exist | Unix              | Windows                                   |
:----------------------------|-------------|---------------------|:------------------|:------------------------------------------|
not set (open existing)      | open        | fail                |                   | OPEN_EXISTING                             |
.create(true)                | open        | create              | O_CREAT           | OPEN_ALWAYS                               |
.truncate(true)              | truncate    | fail                | O_TRUNC           | TRUNCATE_EXISTING                         |
.create(true).truncate(true) | truncate    | create              | O_CREAT + O_TRUNC | CREATE_ALWAYS                             |
.create_new(true)            | fail        | create              | O_CREAT + O_EXCL  | CREATE_NEW + FILE_FLAG_OPEN_REPARSE_POINT |


### Not set (open existing)
Open an existing file. Fails if the file does not exist.


### Create
`.create(true)`

Open an existing file, or create a new file if it does not already exists.


### Truncate
`.truncate(true)`

Open an existing file, and truncate it to zero length. Fails if the file does
not exist. Attributes and permissions of the truncated file are preserved.

Note when using the Windows-specific `.access_mode()`: truncating will only work
if the `GENERIC_WRITE` flag is set. Setting the equivalent individual flags is
not enough.


### Create and truncate
`.create(true).truncate(true)`

Open an existing file and truncate it to zero length, or create a new file if it
does not already exists.

Note when using the Windows-specific `.access_mode()`: Contrary to only
`.truncate(true)`, with `.create(true).truncate(true)` Windows _can_ truncate an
existing file without requiring any flags to be set.

On Windows the attributes of an existing file can cause `.open()` to fail. If
the existing file has the attribute _hidden_ set, it is necessary to open with
`FILE_ATTRIBUTE_HIDDEN`. Similarly if the existing file has the attribute
_system_ set, it is necessary to open with `FILE_ATTRIBUTE_SYSTEM`. See
the Windows-specific `.attributes()` below on how to set these.


### Create_new
`.create_new(true)`

Create a new file, and fail if it already exist.

On Unix this options started its life as a security measure. If you first check
if a file does not exists with `exists()` and then call `open()`, some other
process may have created in the in mean time. `.create_new()` is an atomic
operation that will fail if a file already exist at the location.

`.create_new()` has a special rule on Unix for dealing with symlinks. If there
is a symlink at the final element of its path (e.g. the filename), open will
fail. This is to prevent a vulnerability where an unprivileged process could
trick a privileged process into following a symlink and overwriting a file the
unprivileged process has no access to.
See [Exploiting symlinks and tmpfiles](https://lwn.net/Articles/250468/).
On Windows this behaviour is imitated by specifying not only `CREATE_NEW` but
also `FILE_FLAG_OPEN_REPARSE_POINT`.

Simply put: nothing is allowed to exist on the target location, also no
(dangling) symlink.

if `.create_new(true)` is set, `.create()` and `.truncate()` are ignored.


### Unix-specific: Mode
`.mode(0o666)`

On Unix the new file is created by default with permissions `0o666` minus the
systems `umask` (see [Wikipedia](https://en.wikipedia.org/wiki/Umask)). It is
possible to set on other mode with this option.

If a file already exist or `.create(true)` or `.create_new(true)` are not
specified, `.mode()` is ignored.

Rust currently does not expose a way to modify the umask.


### Windows-specific: Attributes
`.attributes(FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM)`

Files on Windows can have several attributes, most commonly one or more of the
following four: readonly, hidden, system and archive. Most
[others](https://msdn.microsoft.com/en-us/library/windows/desktop/gg258117%28v=vs.85%29.aspx)
are properties set by the file system. Of the others only
`FILE_ATTRIBUTE_ENCRYPTED`, `FILE_ATTRIBUTE_TEMPORARY` and
`FILE_ATTRIBUTE_OFFLINE` can be set when creating a new file. All others are
silently ignored.

It is no use to set the archive attribute, as Windows sets it automatically when
the file is newly created or modified. This flag may then be used by backup
applications as an indication of which files have changed.

If a _new_ file is created because it does not yet exist and `.create(true)` or
`.create_new(true)` are specified, the new file is given the attributes declared
with `.attributes()`.

If an _existing_ file is opened with `.create(true).truncate(true)`, its
existing attributes are preserved and combined with the ones declared with
`.attributes()`.

In all other cases the attributes get ignored.


### Combination of access modes and creation modes

Some combinations of creation modes and access modes do not make sense.

For example: `.create(true)` when opening read-only. If the file does not
already exist, it is created and you start reading from an empty file. And it is
questionable whether you have permission to create a new file if you don't have
write access. A new file is created on all systems I have tested, but there is
no documentation that explicitly guarantees this behaviour.

The same is true for `.truncate(true)` with read and/or append mode. Should an
existing file be modified if you don't have write permission? On Unix it is
undefined
(see [some](http://www.monkey.org/openbsd/archive/tech/0009/msg00299.html)
[comments](http://www.monkey.org/openbsd/archive/tech/0009/msg00304.html) on the
OpenBSD mailing list). The behaviour on Windows is inconsistent and depends on
whether `.create(true)` is set.

To give guarantees about cross-platform (and sane) behaviour, Rust should allow
only the following combinations of access modes and creations modes:

creation mode           | read  | write | read-write | append | read-append |
:-----------------------|:-----:|:-----:|:----------:|:------:|:-----------:|
not set (open existing) |   X   |   X   |     X      |   X    |      X      |
create                  |       |   X   |     X      |   X    |      X      |
truncate                |       |   X   |     X      |        |             |
create and truncate     |       |   X   |     X      |        |             |
create_new              |       |   X   |     X      |   X    |      X      |

It is possible to bypass these restrictions by using system-specific options (as
in this case you already have to take care of cross-platform support yourself).
On Unix this is done by setting the creation mode using `.custom_flags()` with
`O_CREAT`, `O_TRUNC` and/or `O_EXCL`. On Windows this can be done by manually
specifying `.access_mode()` (see above).


## Sharing / locking
On Unix it is possible for multiple processes to read and write to the same file
at the same time.

When you open a file on Windows, the system by default denies other processes to
read or write to the file, or delete it. By setting the sharing mode, it is
possible to allow other processes read, write and/or delete access. For
cross-platform consistency, Rust imitates Unix by setting all sharing flags.

Unix has no equivalent to the kind of file locking that Windows has. It has two
types of advisory locking, POSIX and BSD-style. Advisory means any process that
does not use locking itself can happily ignore the locking af another process.
As if that is not bad enough, they both have
[problems](http://0pointer.de/blog/projects/locking.html) that make them close
to unusable for modern multi-threaded programs. Linux may in some very rare
cases support mandatory file locking, but it is just as broken as advisory.

For Rust, the sharing mode can be set with a Windows-specific option. Given the
problems above, I don't expect there to ever be a cross-platform option for file
locking.


### Windows-specific: Share mode
`.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)`

It is possible to set the individual share permissions with `.share_mode()`.

The current philosophy of this function is that others should have no rights,
unless explicitly granted. I think a better fit for Rust would be to give all
others all rights, unless explicitly denied, e.g.:
`.share_mode(DENY_READ | DENY_WRITE | DENY_DELETE)`.


## Caching behaviour

### Read cache hint
Instead of requesting only the data necessary for a single `read()` call from a
storage device, an operating system may request more data than necessary to have
it already available for the next read call (e.g. the read-ahead cache). If you
read the file sequentially this is beneficial, for completely random access it
can become a penalty. Operating systems generally have good heuristics, but you
may get a performance win if you give the os a hint about how you will read the
file.

Do some real-world benchmarks before setting this option.


#### Cache hint
```
.cache_hint(enum CacheHint)

enum CacheHint {
    None,
    Sequential,
    Random,
}
```

On Windows this maps to the flags `FILE_FLAG_SEQUENTIAL_SCAN` and
`FILE_FLAG_RANDOM_ACCESS`. On Linux and FreeBSD they map to the system call
`posix_fadvise()` with the flags `POSIX_FADV_SEQUENTIAL` and
`POSIX_FADV_RANDOM`.

This option is ignored on operating systems that do not support caching hints.


### Write cache
See [Ensuring data reaches disk](https://lwn.net/Articles/457667/)

1. copy data to kernel space
2. the kernel may wait a short wile
3. data is written to the cache of the storage device
4. data is written to persistent storage

The Rust functions `sync_all()` and `sync_data()` control step 2: they force all
data in the write buffer of the kernel to be written to the storage device. This
is important to ensure critical data reaches the storage device in case of a
system crash or power outage, but comes with a large performance penalty.

All modern operating systems also support a mode where each call to `write()`
will not return until the data is written to the storage device, thus removing
step 2 for _all_ writes. This can be a useful options for writing critical data,
where you would call `sync_data()` after each write. This saves a system call
for each write, and you are sure to never forget it.


#### Sync all
`.sync_all(true)`: implement an open option with the same name as the free
standing call.

On Windows this means setting the flag `FILE_FLAG_WRITE_THROUGH`, and on Unix
(except OS X) the flag `O_SYNC`.

OS X does not support `O_SYNC`, but it is possible to call fcntl with
`F_NOCACHE` to get the same effect. This has the side-effect that data also does
not end up in the read cache, so this can have a performance penalty when
reading if a file is opened with a read-write access mode.


#### Sync data
`.sync_data(true)`

Some systems support syncing only the data written, but can wait with updating
less critical metadata such as the last modified timestamp. If the metadata is
not critical (and it rarely is), you should always use `sync_data()` as an easy
performance win.

Linux since version 2.6.33 supports this mode with `O_DSYNC`, as does Solaris
and recent versions of NetBSD. If a system does not support only syncing data,
Rust will fall back to full syncing.

If `.sync_all(true)` is specified, `.sync_data()` is ignored.


### Completely bypass the kernel
Normally the os kernel will process read or write calls and store the data
temporarily in a kernel-space buffer. The kernel makes sure the data size and
alignment of reads and writes correspondent to sectors on the storage device,
usually 512 or 4096 bytes. Also the kernel can keep data recently read or
written in cache, to speed up future file operations.

Some operating systems allow you to completely bypass the copy of data to or
from kernel space. This is generally a bad idea. Applications will have to
figure out and handle alignment restrictions themselves, and implement manual
caching. It is mostly useful for database applications that may have more
knowledge about their optimal caching behaviour than the os. And it can have a
use when reading many gigabytes of data (like a backup process), which may
destroy the os cache for other processes.

This is available on Windows with the flag `FILE_FLAG_NO_BUFFERING`, and on
Linux and some variants of BSD with `O_DIRECT`. Making correct use of this mode
involves low-level tuning and operating system dependant behaviour. It makes no
sense for Rust to expose this as a simple, cross-platform option. For
applications that really wish to use it, it is no problem to submit it as a
custom flag.


## Asynchronous IO
Out op scope.


## Other options

### Inheritance of file descriptors
Leaking file descriptors to child processes can cause problems and can be a
security vulnerability. See this report by 
[Python](https://www.python.org/dev/peps/pep-0446/).

On Windows, child processes do not inherit file descriptors by default (but this
can be changed). On Unix they always inherit, unless the close-on-exec flag is
set.

The close on exec flag can be set atomically when opening the file, or later
with `fcntl`. The `O_CLOEXEC` flag is in the relatively new POSIX-2008 standard,
and all modern versions of Unix support it. The following table lists for which
operating systems we can rely on the flag to be supported.

os            | since version | oldest supported version
:-------------|:--------------|:------------------------
OS X          | 10.6          | 10.7?
Linux         | 2.6.23        | 2.6.32 (supported by Rust)
FreeBSD       | 8.3           | 8.4
OpenBSD       | 5.0           | 5.7
NetBSD        | 6.0           | 5.0
Dragonfly BSD | 3.2           | ? (3.2 is not updated since 2012-12-14)
Solaris       | 11            | 10

This means we can always set the flag `O_CLOEXEC`, and do an additional `fcntl`
if the os is NetBSD or Solaris.


### Custom flags
`.custom_flags()`

Windows and the various flavours of Unix support flags that are not
cross-platform, but that can be useful in some circumstances. On Unix they will
be passed as the variable _flags_ to `open`, on Windows as the
_dwFlagsAndAttributes_ parameter.

The cross-platform options of Rust can do magic: they can set any flag necessary
to ensure it works as expected. For example, `.append(true)` on Unix not only
sets the flag `O_APPEND`, but also automatically `O_WRONLY` or `O_RDWR`. This
special treatment is not available for the custom flags.

Custom flags can only set flags, not remove flags set by Rusts options.

For the custom flags on Unix, the bits that define the access mode are masked
out with `O_ACCMODE`, to ensure they do not interfere with the access mode set
by Rusts options.

| [Windows](https://msdn.microsoft.com/en-us/library/windows/desktop/hh449426%28v=vs.85%29.aspx):
|:---------------------------
| FILE_FLAG_BACKUP_SEMANTICS
| FILE_FLAG_DELETE_ON_CLOSE
| FILE_FLAG_NO_BUFFERING
| FILE_FLAG_OPEN_NO_RECALL
| FILE_FLAG_OPEN_REPARSE_POINT
| FILE_FLAG_OVERLAPPED
| FILE_FLAG_POSIX_SEMANTICS
| FILE_FLAG_RANDOM_ACCESS
| FILE_FLAG_SESSION_AWARE
| FILE_FLAG_SEQUENTIAL_SCAN
| FILE_FLAG_WRITE_THROUGH

Unix:

| POSIX       | Linux       | OS X        | FreeBSD     | OpenBSD     | NetBSD      |Dragonfly BSD| Solaris     |
|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|
| O_DIRECTORY | O_DIRECTORY |             | O_DIRECTORY | O_DIRECTORY | O_DIRECTORY | O_DIRECTORY | O_DIRECTORY |
| O_NOCTTY    | O_NOCTTY    |             | O_NOCTTY    |             | O_NOCTTY    |             | O_NOCTTY    |
| O_NOFOLLOW  | O_NOFOLLOW  | O_NOFOLLOW  | O_NOFOLLOW  | O_NOFOLLOW  | O_NOFOLLOW  | O_NOFOLLOW  | O_NOFOLLOW  |
| O_NONBLOCK  | O_NONBLOCK  | O_NONBLOCK  | O_NONBLOCK  | O_NONBLOCK  | O_NONBLOCK  | O_NONBLOCK  | O_NONBLOCK  |
| O_DSYNC     | O_DSYNC     |             |             |             | O_DSYNC     |             | O_DSYNC     |
| O_RSYNC     |             |             |             |             | O_RSYNC     |             | O_RSYNC     |
| O_SYNC      | O_SYNC      |             | O_SYNC      | O_SYNC      | O_SYNC      | O_FSYNC     | O_SYNC      |
|             | O_DIRECT    |             | O_DIRECT    |             | O_DIRECT    | O_DIRECT    |             |
|             | O_ASYNC     |             |             |             | O_ASYNC     |             |             |
|             | O_NOATIME   |             |             |             |             |             |             |
|             | O_PATH      |             |             |             |             |             |             |
|             | O_TMPFILE   |             |             |             |             |             |             |
|             |             | O_SHLOCK    | O_SHLOCK    | O_SHLOCK    | O_SHLOCK    | O_SHLOCK    |             |
|             |             | O_EXLOCK    | O_EXLOCK    | O_EXLOCK    | O_EXLOCK    | O_EXLOCK    |             |
|             |             | O_SYMLINK   |             |             |             |             |             |
|             |             | O_EVTONLY   |             |             |             |             |             |
|             |             |             |             |             | O_NOSIGPIPE |             |             |
|             |             |             |             |             | O_ALT_IO    |             |             |
|             |             |             |             |             |             |             | O_NOLINKS   |
|             |             |             |             |             |             |             | O_XATTR     |
| [POSIX](http://pubs.opengroup.org/onlinepubs/9699919799/functions/open.html) | [Linux](http://man7.org/linux/man-pages/man2/open.2.html) | [OS X](https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man2/open.2.html) | [FreeBSD](https://www.freebsd.org/cgi/man.cgi?query=open&sektion=2) | [OpenBSD](http://www.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man2/open.2?query=open&sec=2) | [NetBSD](http://netbsd.gw.com/cgi-bin/man-cgi?open+2+NetBSD-current) | [Dragonfly BSD](http://leaf.dragonflybsd.org/cgi/web-man?command=open&section=2) | [Solaris](http://docs.oracle.com/cd/E23824_01/html/821-1463/open-2.html) |


### Windows-specific flags and attributes
The following variables for CreateFile2 currently have no equivalent functions
in Rust to set them:
```
DWORD                 dwSecurityQosFlags;
LPSECURITY_ATTRIBUTES lpSecurityAttributes;
HANDLE                hTemplateFile;
```


## Changes from current

### Access mode
- Current: `.append(true)` requires `.write(true)` on Unix, but not on Windows.
  New: ignore `.write()` if `.append(true)` is specified.
- Current: when `.append(true)` is set, it is not possible to modify file
  attributes on Windows, but it is possible to change the file mode on Unix.
  New: allow file attributes to be modified on Windows in append-mode.
- Current: `.read()` and `.write()` set individual bit flags instead of generic
  flags. New: Set generic flags, as recommend by Microsoft. e.g. `GENERIC_WRITE`
  instead of `FILE_GENERIC_WRITE` and `GENERIC_READ` instead of
  `FILE_GENERIC_READ`. Currently truncate is broken on Windows, this fixes it.
- Current: when no access mode is set, this falls back to opening the file
  read-only on Unix.
  New: open with `O_RDONLY | O_PATH` on Linux, and fail with `E_INVALID` on all
  other Unix variants.
- Rename the Windows-specific `.desired_access()` to `.access_mode()`

### Creation mode
- Do not allow `.truncate(true)` if the access mode is read-only and/or append.
  This is currently buggy on Windows, and works on some versions of Unix, but
  not on others (implementation defined).
- Implement `.create_new()`.
- Remove the Windows-specific `.creation_disposition()`.
  It has no use, because all its options can be set in a cross-platform way.
- Split the Windows-specific `.flags_and_attributes()` into `.custom_flags()`
  and `.attributes()`. This is a form of future-proofing, as the new Windows 8
  `Createfile2` also splits these attributes. This has the advantage of a clear
  separation between file attributes, that are somewhat similar to Unix mode
  bits, and the custom flags that modify the behaviour of the current file
  handle.

### Sharing / locking
- Currently `.share_mode()` grants permissions, change it to grant by default,
  and possibly deny permissions.

### Caching behaviour
- Implement `.cache_hint()`.
- Implement `.sync_all()` and `.sync_data()`.

### Other options
- Set the close-on-exec flag atomically on Unix if supported.
- Implement `.custom_flags()` on Windows and Unix to pass custom flags to the
system.


# Drawbacks
This adds a thin layer on top of the raw operating system calls. In this
[pull request](https://github.com/rust-lang/rust/pull/26772#issuecomment-126753342)
the conclusion was: this seems like a good idea for a "high level" abstraction
like OpenOptions.

This adds extra options that many applications can do without (otherwise they
were already implemented).

Also this RFC is in line with the vision for IO in the
[IO-OS-redesign](https://github.com/rust-lang/rfcs/blob/master/text/0517-io-os-reform.md#vision-for-io):
- [The APIs] should impose essentially zero cost over the underlying OS
  services; the core APIs should map down to a single syscall unless more are
  needed for cross-platform compatibility.
- The APIs should largely feel like part of "Rust" rather than part of any
  legacy, and they should enable truly portable code.
- Coverage. The std APIs should over time strive for full coverage of non-niche,
  cross-platform capabilities.


# Alternatives
Keep the status quo.


# Unresolved questions
Implementation and testing of `.sync_all()` and `.sync_data()` could uncover
some corner cases, but I don't expect any that would give great trouble.

Should `.cache_hint()` take an enum?

Rename the Windows-specific `.desired_access()` to `.access_mode()`?

What should be done about the missing variables for `CreateFile2`?

Are there any other options that we should define while at it?