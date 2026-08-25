//Error output test for #78834: Type is too big for the target architecture
struct B<
    A: Sized = [(); {
                   let x = [0u8; !0usize];
                   //~^ ERROR too big for the target architecture
                   1
               }],
> {
    a: A,
}

fn main() {}
