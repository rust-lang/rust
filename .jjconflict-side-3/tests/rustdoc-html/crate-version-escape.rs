//@ compile-flags: --crate-version=<script>alert("hi")</script> -Z unstable-options

#![crate_name = "foo"]

//@ has 'foo/index.html' '//*[@class="version"]' '<script>alert("hi")</script>'
