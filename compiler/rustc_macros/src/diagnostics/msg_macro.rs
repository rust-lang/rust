use syn::{LitStr, parse_macro_input};

use crate::diagnostics::message::Message;

pub(crate) fn msg_macro(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let inline = parse_macro_input!(input as LitStr);
    let message =
        Message { attr_span: inline.span(), message_span: inline.span(), value: inline.value() };
    message.diag_message(None).into()
}
