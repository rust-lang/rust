#![feature(restricted_std)]

use std::collections::BTreeMap;
use std::task::{Context, Poll};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Default)]
pub struct ChatRequest {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    // Optional metadata bag for clients and intermediaries.
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Canceled,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatDelta {
    pub text: String,
    pub finish: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmError {
    Canceled,
    Timeout,
    Protocol(String),
    Decode(String),
    Transport(String),
    Other(String),
}

pub trait ChatStream: Send {
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<ChatDelta>, LlmError>>;
}

pub trait StreamingLlmClient {
    fn chat_stream(&self, req: ChatRequest) -> Result<Box<dyn ChatStream + Send>, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{RawWaker, RawWakerVTable, Waker};

    fn noop_waker() -> Waker {
        unsafe fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(core::ptr::null(), &VTABLE)
        }
        unsafe fn wake(_: *const ()) {}
        unsafe fn wake_by_ref(_: *const ()) {}
        unsafe fn drop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    struct TestStream {
        idx: usize,
    }

    impl ChatStream for TestStream {
        fn poll_next(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<Option<ChatDelta>, LlmError>> {
            let item = match self.idx {
                0 => Some("h"),
                1 => Some("i"),
                _ => None,
            };
            self.idx += 1;
            let out = item.map(|text| ChatDelta {
                text: text.into(),
                finish: None,
            });
            Poll::Ready(Ok(out))
        }
    }

    #[test]
    fn consumer_reads_stream_to_completion() {
        let mut stream = TestStream { idx: 0 };
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut out = String::new();

        loop {
            match stream.poll_next(&mut cx) {
                Poll::Ready(Ok(Some(delta))) => out.push_str(&delta.text),
                Poll::Ready(Ok(None)) => break,
                Poll::Ready(Err(err)) => panic!("unexpected error: {:?}", err),
                Poll::Pending => continue,
            }
        }

        assert_eq!(out, "hi");
    }
}
