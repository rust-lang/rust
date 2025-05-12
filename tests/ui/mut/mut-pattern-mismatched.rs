//@ dont-require-annotations: NOTE

fn main() {
    let foo = &mut 1;

    // (separate lines to ensure the spans are accurate)

     let &_ //~  ERROR mismatched types
            //~| NOTE expected mutable reference `&mut {integer}`
            //~| NOTE found reference `&_`
            //~| NOTE types differ in mutability
        = foo;
    let &mut _ = foo;

    let bar = &1;
    let &_ = bar;
    let &mut _ //~  ERROR mismatched types
               //~| NOTE expected reference `&{integer}`
               //~| NOTE found mutable reference `&mut _`
               //~| NOTE types differ in mutability
         = bar;
}
