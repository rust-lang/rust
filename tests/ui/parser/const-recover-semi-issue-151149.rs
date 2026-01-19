#![feature(const_trait_impl)]

const trait ConstDefault {
    fn const_default() -> Self;
}

impl const ConstDefault for u8 {
    fn const_default() -> Self { 0 }
}

const fn val() -> u8 {
    42
}

const C: u8 = u8::const_default()
&1 //~ ERROR expected `;`, found keyword `const`

const fn foo() -> &'static u8 {
    const C: u8 = u8::const_default() //~ ERROR expected `;`
    &C
}

const fn bar() -> u8 { //~ ERROR mismatched types
    const C: u8 = 1
     + 2 //~ ERROR expected `;`, found `}`
}

const fn baz() -> u8 { //~ ERROR mismatched types
    const C: u8 = 1
     + val() //~ ERROR expected `;`, found `}`
}

fn buzz() -> &'static u8 {
    let r = 1 //~ ERROR expected `;`
    &r
}

fn main() {}
