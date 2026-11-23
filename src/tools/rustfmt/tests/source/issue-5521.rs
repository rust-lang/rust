// Whitespace on the otherwise blank line before a `#[rustfmt::skip]` item has to be
// removed regardless of how many whitespace characters it is made of.
impl Even for Foo {
    const FAIL: &'static str = "";
    
    #[rustfmt::skip]
    fn to_style(&self, template: &str) {}
}

impl Odd for Foo {
    const FAIL: &'static str = "";
   
    #[rustfmt::skip]
    fn to_style(&self, template: &str) {}
}

impl Single for Foo {
    const FAIL: &'static str = "";
 
    #[rustfmt::skip]
    fn to_style(&self, template: &str) {}
}

impl Tabs for Foo {
    const FAIL: &'static str = "";
		
    #[rustfmt::skip]
    fn to_style(&self, template: &str) {}
}
