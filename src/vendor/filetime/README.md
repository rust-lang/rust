# filetime

[![Build Status](https://travis-ci.org/alexcrichton/filetime.svg?branch=master)](https://travis-ci.org/alexcrichton/filetime)
[![Build status](https://ci.appveyor.com/api/projects/status/9tatexq47i3ee13k?svg=true)](https://ci.appveyor.com/project/alexcrichton/filetime)

[Documentation](http://alexcrichton.com/filetime/filetime/index.html)

A helper library for inspecting the various timestamps of files in Rust. This
library takes into account cross-platform differences in terms of where the
timestamps are located, what they are called, and how to convert them into a
platform-independent representation.

```toml
# Cargo.toml
[dependencies]
filetime = "0.1"
```

# License

`filetime` is primarily distributed under the terms of both the MIT license and
the Apache License (Version 2.0), with portions covered by various BSD-like
licenses.

See LICENSE-APACHE, and LICENSE-MIT for details.
