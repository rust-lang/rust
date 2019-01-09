// gate-test-concat_idents

fn main() {
    concat_idents!(a, b); //~ ERROR `concat_idents` is not stable enough
                          //~| ERROR cannot find value `ab` in this scope
}
