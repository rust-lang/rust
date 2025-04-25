#![expect(deprecated)] // concat_idents is deprecated

fn main() {
    concat_idents!(a, b); //~ ERROR `concat_idents` is not stable enough
                          //~| ERROR cannot find value `ab` in this scope
}
