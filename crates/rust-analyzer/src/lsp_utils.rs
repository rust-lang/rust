//! Utilities for LSP-related boilerplate code.

use crossbeam_channel::Sender;
use lsp_server::{Message, Notification, Request, RequestId};
use ra_db::Canceled;
use serde::{de::DeserializeOwned, Serialize};
use std::error::Error;

pub fn show_message(
    typ: lsp_types::MessageType,
    message: impl Into<String>,
    sender: &Sender<Message>,
) {
    let message = message.into();
    let params = lsp_types::ShowMessageParams { typ, message };
    let not = notification_new::<lsp_types::notification::ShowMessage>(params);
    sender.send(not.into()).unwrap();
}

pub(crate) fn is_canceled(e: &(dyn Error + 'static)) -> bool {
    e.downcast_ref::<Canceled>().is_some()
}

pub(crate) fn notification_is<N: lsp_types::notification::Notification>(
    notification: &Notification,
) -> bool {
    notification.method == N::METHOD
}

pub(crate) fn notification_cast<N>(notification: Notification) -> Result<N::Params, Notification>
where
    N: lsp_types::notification::Notification,
    N::Params: DeserializeOwned,
{
    notification.extract(N::METHOD)
}

pub(crate) fn notification_new<N>(params: N::Params) -> Notification
where
    N: lsp_types::notification::Notification,
    N::Params: Serialize,
{
    Notification::new(N::METHOD.to_string(), params)
}

pub(crate) fn request_new<R>(id: RequestId, params: R::Params) -> Request
where
    R: lsp_types::request::Request,
    R::Params: Serialize,
{
    Request::new(id, R::METHOD.to_string(), params)
}
