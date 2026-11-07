fn main() {
    let _ = 
r#"
this is a very long string exceeded maximum width in this case maximum 100. (current this line width is about 115)
"#;

    let _with_newline = 
    
r#"
this is a very long string exceeded maximum width in this case maximum 100. (current this line width is about 115)
"#;
}
