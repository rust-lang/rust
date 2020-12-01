use crate::spec::Target;

pub fn target() -> Target {
    super::avr_gnu_base::target("atmega328".to_owned())
}
