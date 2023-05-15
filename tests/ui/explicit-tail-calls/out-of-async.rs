// edition: 2021
#![feature(explicit_tail_calls)]

async fn a() {
    become b();
    //~^ error: mismatched function ABIs
    //~| error: mismatched signatures
}

fn b() {}

fn block() -> impl std::future::Future<Output = ()> {
    async {
        become b();
        //~^ error: mismatched function ABIs
        //~| error: mismatched signatures
    }
}

fn main() {}
