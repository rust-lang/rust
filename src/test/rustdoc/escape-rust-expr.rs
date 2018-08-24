// Test that we HTML-escape Rust expressions, where HTML special chars
// can occur, and we know it's definitely not markup.

// @has escape_rust_expr/constant.CONST_S.html '//pre[@class="rust const"]' '"<script>"'
pub const CONST_S: &'static str = "<script>";
