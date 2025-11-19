#!/path/to/somewhere 0 if false ""

// Test that we highlight shebang as comments on source code pages.

//@ has 'src/shebang/shebang.rs.html'
//@ has - '//pre[@class="rust"]//span[@class="comment"]' '#!/path/to/somewhere 0 if false ""'
