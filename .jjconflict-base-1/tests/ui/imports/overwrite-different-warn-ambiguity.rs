//@ check-pass
//@ edition:2024

mod framing {
    mod public_message_in {
        mod public_message {
            mod public_message {
                pub struct ConfirmedTranscriptHashInput;
            }
            mod public_message_in {
                use super::*;
                #[derive(Debug)]
                pub struct ConfirmedTranscriptHashInput;
            }
            pub use public_message::*;
            use public_message_in::*;
        }
        mod public_message_in {
            #[derive(Debug)]
            pub struct ConfirmedTranscriptHashInput;
        }
        pub use public_message::*;
        use public_message_in::*;
    }
    use public_message_in::*;
}

fn main() {}
