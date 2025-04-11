fn main() {
    let foo = &mut 1;

    // (separate lines to ensure the spans are accurate)

     let &_ //~  ERROR mismatched types
            //~| NOTE_NONVIRAL expected mutable reference `&mut {integer}`
            //~| NOTE_NONVIRAL found reference `&_`
            //~| NOTE_NONVIRAL types differ in mutability
        = foo;
    let &mut _ = foo;

    let bar = &1;
    let &_ = bar;
    let &mut _ //~  ERROR mismatched types
               //~| NOTE_NONVIRAL expected reference `&{integer}`
               //~| NOTE_NONVIRAL found mutable reference `&mut _`
               //~| NOTE_NONVIRAL types differ in mutability
         = bar;
}
