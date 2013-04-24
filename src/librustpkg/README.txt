Right now, commands that work are "build" and "clean".

`rustpkg build` and `rustpkg clean` should work; for example:
$ cd ~/rust/src/librustpkg/testsuite/pass
$ rustpkg build hello-world
... some output ...
$ rustpkg clean hello-world

rustpkg works correctly with the following test packages in librustpkg/testsuite/pass:
      * hello-world
      * install-paths
      * simple-lib
      * deeply/nested/path
      * fancy-lib

Right now, it fails on the following test packages:
      * external-crate (no support for `extern mod` inference yet)

It correctly rejects the test packages in librustpkg/testsuite/fail with the
expected error messages.
