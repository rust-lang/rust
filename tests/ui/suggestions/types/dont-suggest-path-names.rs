// This is a regression test for #123630
//
// Prior to #123703 this was resulting in compiler suggesting add a type signature
// for `lit` containing path to a file containing `Select` - something obviously invalid.

struct Select<F, I>(F, I);
fn select<F, I>(filter: F) -> Select<F, I> {}
//~^ 7:31: 7:43: mismatched types [E0308]

fn parser1() {
    let lit = select(|x| match x {
        //~^ 11:23: 11:24: type annotations needed [E0282]
        _ => (),
    });
}

fn main() {}
