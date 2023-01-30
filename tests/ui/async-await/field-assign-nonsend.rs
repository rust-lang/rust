// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// Derived from an ICE found in tokio-xmpp during a crater run.
// edition:2021

#![allow(dead_code)]

#[derive(Clone)]
struct InfoResult {
    node: Option<std::rc::Rc<String>>
}

struct Agent {
    info_result: InfoResult
}

impl Agent {
    async fn handle(&mut self) {
        let mut info = self.info_result.clone();
        info.node = None;
        let element = parse_info(info);
        let _ = send_element(element).await;
    }
}

struct Element {
}

async fn send_element(_: Element) {}

fn parse(_: &[u8]) -> Result<(), ()> {
    Ok(())
}

fn parse_info(_: InfoResult) -> Element {
    Element { }
}

fn assert_send<T: Send>(_: T) {}

fn main() {
    let agent = Agent { info_result: InfoResult { node: None } };
    // FIXME: It would be nice for this to work. See #94067.
    assert_send(agent.handle());
    //~^ cannot be sent between threads safely
}
