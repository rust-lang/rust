// https://github.com/rust-lang/rust/pull/113099

mod framing {
    mod public_message {
        use super::*;

        #[derive(Debug)]
        pub struct ConfirmedTranscriptHashInput;
    }

    mod public_message_in {
        use super::*;

        #[derive(Debug)]
        pub struct ConfirmedTranscriptHashInput;
    }

    pub use self::public_message::*;
    pub use self::public_message_in::*;
}

use crate::framing::ConfirmedTranscriptHashInput;
//~^ ERROR `ConfirmedTranscriptHashInput` is ambiguous

fn main() { }
