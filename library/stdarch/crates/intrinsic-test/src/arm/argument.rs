use crate::arm::intrinsic::ArmIntrinsicType;
use crate::common::argument::Argument;

// This functionality is present due to the nature
// of how intrinsics are defined in the JSON source
// of ARM intrinsics.
impl Argument<ArmIntrinsicType> {
    pub fn type_and_name_from_c(arg: &str) -> (&str, &str) {
        let split_index = arg
            .rfind([' ', '*'])
            .expect("Couldn't split type and argname");

        (arg[..split_index + 1].trim_end(), &arg[split_index + 1..])
    }
}
