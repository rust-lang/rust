//@ run-rustfix

struct TargetStruct;

impl From<usize> for TargetStruct {
    fn from(_unchecked: usize) -> Self {
        TargetStruct
    }
}

fn main() {
    let a = &3;
    let _b: TargetStruct = a.into(); //~ ERROR the trait bound `TargetStruct: From<&{integer}>` is not satisfied
}
