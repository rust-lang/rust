// ignore-tidy-tab

fn main() {
    format!("{
    a");
    //~^ ERROR invalid format string
    format!("{ \
               \
    b");
    //~^ ERROR invalid format string
    format!(r#"{ \

    rawc"#);
    //~^^^ ERROR invalid format string
    format!(r#"{ \n
\n
    rawd"#);
    //~^^^ ERROR invalid format string
    format!("{ \n
\n
    e");
    //~^ ERROR invalid format string
    format!("
    {
    a");
    //~^ ERROR invalid format string
    format!("
    {
    a
    ");
    //~^^ ERROR invalid format string
    format!("  \
    { \
    	\
    b");
    //~^ ERROR invalid format string
    format!("  \
    { \
    	\
    b \
      \
    ");
    //~^^^ ERROR invalid format string
    format!(r#"
raw  { \
       \
    c"#);
    //~^^^ ERROR invalid format string
    format!(r#"
raw  { \n
\n
    d"#);
    //~^^^ ERROR invalid format string
    format!("
  { \n
\n
    e");
    //~^ ERROR invalid format string

    format!("
    {asdf
    }
    ", asdf=1);
    // ok - this is supported
    format!("
    {
    asdf}
    ", asdf=1);
    //~^^ ERROR invalid format string
    println!("\t{}");
    //~^ ERROR 1 positional argument in format string

    // note: `\x7B` is `{`
    println!("\x7B}\u{8} {", 1);
    //~^ ERROR invalid format string: expected `}` but string was terminated

    println!("\x7B}\u8 {", 1);
    //~^ ERROR incorrect unicode escape sequence

    // note: raw strings don't escape `\xFF` and `\u{FF}` sequences
    println!(r#"\x7B}\u{8} {"#, 1);
    //~^ ERROR invalid format string: unmatched `}` found

    println!(r#"\x7B}\u8 {"#, 1);
    //~^ ERROR invalid format string: unmatched `}` found
}
