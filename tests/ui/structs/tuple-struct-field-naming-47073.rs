// https://github.com/rust-lang/rust/issues/47073
type Guilty = bool;
type FineDollars = u32;

struct Verdict(Guilty, Option<FineDollars>);

fn main() {
    let justice = Verdict(true, Some(2718));
    let _condemned = justice.00;
    //~^ ERROR no field `00` on type `Verdict`
    let _punishment = justice.001;
    //~^ ERROR no field `001` on type `Verdict`
}
