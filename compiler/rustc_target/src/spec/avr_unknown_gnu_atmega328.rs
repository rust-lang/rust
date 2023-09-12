use crate::spec::{base::avr_gnu, Target};

pub fn target() -> Target {
    avr_gnu::target("atmega328", "-mmcu=atmega328")
}
