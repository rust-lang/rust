use crate::spec::{base, Target};

pub fn target() -> Target {
    base::avr_gnu::target("atmega328")
}
