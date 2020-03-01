// compile-flags: --crate-version=<script>alert("hi")</script> -Z unstable-options

#![crate_name = "foo"]

// @has 'foo/index.html' '//div[@class="block version"]/p' 'Version <script>alert("hi")</script>'
// @has 'foo/all.html' '//div[@class="block version"]/p' 'Version <script>alert("hi")</script>'
