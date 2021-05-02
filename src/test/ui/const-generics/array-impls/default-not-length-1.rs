struct NotDefault;

fn main() {
    let _: [NotDefault; 1] = Default::default();
    //~^ ERROR the trait bound `NotDefault: Default` is not satisfied
}
