use crate::spec::{base, Target};

pub(crate) fn target() -> Target {
    base::avr_gnu::target("atmega328", "-mmcu=atmega328")
}
