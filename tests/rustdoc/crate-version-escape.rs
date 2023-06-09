// compile-flags: --crate-version=<script>alert("hi")</script> -Z unstable-options

#![crate_name = "foo"]

// @has 'foo/index.html' '//li[@class="version"]' 'Version <script>alert("hi")</script>'
