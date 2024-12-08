use crate::spec::{Target, base};

pub(crate) fn target() -> Target {
    base::avr_gnu::target("atmega328", "-mmcu=atmega328")
}
