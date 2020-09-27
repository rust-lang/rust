use crate::spec::TargetResult;

pub fn target() -> TargetResult {
    super::avr_gnu_base::target("atmega328".to_owned())
}
