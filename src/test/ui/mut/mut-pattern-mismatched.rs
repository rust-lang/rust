fn main() {
    let foo = &mut 1;

    // (separate lines to ensure the spans are accurate)

     let &_ //~  ERROR mismatched types
            //~| expected type `&mut {integer}`
            //~| found type `&_`
            //~| types differ in mutability
        = foo;
    let &mut _ = foo;

    let bar = &1;
    let &_ = bar;
    let &mut _ //~  ERROR mismatched types
               //~| expected type `&{integer}`
               //~| found type `&mut _`
               //~| types differ in mutability
         = bar;
}
