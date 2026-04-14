#![feature(restricted_std)]

use std::task::{Context, Poll};

use llm::{ChatDelta, ChatRequest, ChatStream, FinishReason, LlmError, StreamingLlmClient};

pub struct StubLlmClient;

impl StubLlmClient {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for StubLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingLlmClient for StubLlmClient {
    fn chat_stream(&self, _req: ChatRequest) -> Result<Box<dyn ChatStream + Send>, LlmError> {
        Ok(Box::new(StubChatStream::new()))
    }
}

struct StubChatStream {
    step: u8,
}

impl StubChatStream {
    fn new() -> Self {
        Self { step: 0 }
    }

    fn delta(text: &str, finish: Option<FinishReason>) -> ChatDelta {
        ChatDelta {
            text: String::from(text),
            finish,
        }
    }
}

impl ChatStream for StubChatStream {
    fn poll_next(&mut self, _cx: &mut Context<'_>) -> Poll<Result<Option<ChatDelta>, LlmError>> {
        let out = match self.step {
            0 => {
                self.step = 1;
                Some(Self::delta("Som", None))
            }
            1 => {
                self.step = 2;
                Some(Self::delta("eth", None))
            }
            2 => {
                self.step = 3;
                Some(Self::delta("ing", Some(FinishReason::Stop)))
            }
            _ => None,
        };
        Poll::Ready(Ok(out))
    }
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

    #[test]
    fn stub_streams_three_chunks_in_order() {
        let client = StubLlmClient::new();
        let mut stream = client.chat_stream(ChatRequest::default()).expect("stream");
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut chunks = Vec::new();
        loop {
            match stream.poll_next(&mut cx) {
                Poll::Ready(Ok(Some(delta))) => chunks.push(delta.text),
                Poll::Ready(Ok(None)) => break,
                Poll::Ready(Err(err)) => panic!("unexpected error: {:?}", err),
                Poll::Pending => continue,
            }
        }

        assert_eq!(chunks, ["Som", "eth", "ing"]);
    }

    #[test]
    fn stub_finishes_with_stop() {
        let client = StubLlmClient::new();
        let mut stream = client.chat_stream(ChatRequest::default()).expect("stream");
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut last_finish = None;
        loop {
            match stream.poll_next(&mut cx) {
                Poll::Ready(Ok(Some(delta))) => last_finish = delta.finish,
                Poll::Ready(Ok(None)) => break,
                Poll::Ready(Err(err)) => panic!("unexpected error: {:?}", err),
                Poll::Pending => continue,
            }
        }

        assert_eq!(last_finish, Some(FinishReason::Stop));
    }
}
