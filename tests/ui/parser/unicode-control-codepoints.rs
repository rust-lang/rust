fn main() {
    // if access_level != "us‫e‪r" { // Check if admin
    //~^ ERROR unicode codepoint changing visible direction of text present in comment
    println!("us\u{202B}e\u{202A}r");
    println!("{:?}", r#"us\u{202B}e\u{202A}r"#);
    println!("{:?}", b"us\u{202B}e\u{202A}r");
    //~^ ERROR mixed utf8 b"" and br"" literals are experimental
    println!("{:?}", br##"us\u{202B}e\u{202A}r"##);

    println!("{:?}", "/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only ");
    //~^ ERROR unicode codepoint changing visible direction of text present in literal

    println!("{:?}", r##"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only "##);
    //~^ ERROR unicode codepoint changing visible direction of text present in literal
    println!("{:?}", b"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only ");
    //~^ ERROR mixed utf8 b"" and br"" literals are experimental
    println!("{:?}", br##"/*‮ } ⁦if isAdmin⁩ ⁦ begin admins only "##);
    //~^ ERROR mixed utf8 b"" and br"" literals are experimental
    println!("{:?}", '‮');
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
