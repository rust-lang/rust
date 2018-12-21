// ignore-tidy-tab

fn main() {
    format!("{
    a");
    //~^ ERROR invalid format string
    format!("{ \

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

    ");
    //~^^^ ERROR invalid format string
    format!(r#"
raw  { \

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
}
