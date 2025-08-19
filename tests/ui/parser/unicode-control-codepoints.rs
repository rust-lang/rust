//@ edition: 2021

fn main() {
    // if access_level != "us‫e‪r" { // Check if admin
    //~^ ERROR unicode codepoint changing visible direction of text present in comment
    println!("us\u{202B}e\u{202A}r");
    println!("{:?}", r#"us\u{202B}e\u{202A}r"#);
    println!("{:?}", b"us\u{202B}e\u{202A}r");
    //~^ ERROR unicode escape in byte string
    //~| ERROR unicode escape in byte string
    println!("{:?}", br##"us\u{202B}e\u{202A}r"##);

    println!("{:?}", "/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only ");
    //~^ ERROR unicode codepoint changing visible direction of text present in literal

    println!("{:?}", r##"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only "##);
    //~^ ERROR unicode codepoint changing visible direction of text present in literal
    println!("{:?}", b"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only ");
    //~^ ERROR non-ASCII character in byte string literal
    //~| ERROR non-ASCII character in byte string literal
    //~| ERROR non-ASCII character in byte string literal
    //~| ERROR non-ASCII character in byte string literal
    println!("{:?}", br##"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only "##);
    //~^ ERROR non-ASCII character in raw byte string literal
    //~| ERROR non-ASCII character in raw byte string literal
    //~| ERROR non-ASCII character in raw byte string literal
    //~| ERROR non-ASCII character in raw byte string literal
    println!("{:?}", '‮');
    //~^ ERROR unicode codepoint changing visible direction of text present in literal

    let _ = c"‮";
    //~^ ERROR unicode codepoint changing visible direction of text present in literal
    let _ = cr#"‮"#;
    //~^ ERROR unicode codepoint changing visible direction of text present in literal

    println!("{{‮}}");
    //~^ ERROR unicode codepoint changing visible direction of text present in literal
}

//"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only */"
//~^ ERROR unicode codepoint changing visible direction of text present in comment

/**  '‮'); */fn foo() {}
//~^ ERROR unicode codepoint changing visible direction of text present in doc comment

/**
 *
 *  '‮'); */fn bar() {}
//~^^^ ERROR unicode codepoint changing visible direction of text present in doc comment
