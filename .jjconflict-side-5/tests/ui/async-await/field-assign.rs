// Derived from an ICE found in tokio-xmpp during a crater run.
//@ edition:2021
//@ build-pass

#![allow(dead_code)]

#[derive(Clone)]
struct InfoResult {
    node: Option<String>
}

struct Agent {
    info_result: InfoResult
}

impl Agent {
    async fn handle(&mut self) {
        let mut info = self.info_result.clone();
        info.node = Some("bar".into());
        let element = parse_info(info);
        send_element(element).await;
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

fn main() {
    let mut agent = Agent {
        info_result: InfoResult { node: None }
    };
    let _ = agent.handle();
}
