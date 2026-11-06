//@ rustfmt-style_edition: 2024

mod example {
    pub const CUSTOM_ERROR_MSG_BUFFER_SIZE: usize = 4096; // one kernel page?!
    //^ must be pub due to being used in a macro!
}
