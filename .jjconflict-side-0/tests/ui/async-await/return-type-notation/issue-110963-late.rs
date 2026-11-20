//@ edition: 2021
//@ check-pass

#![feature(return_type_notation)]

trait HealthCheck {
    async fn check(&mut self) -> bool;
}

async fn do_health_check_par<HC>(hc: HC)
where
    HC: HealthCheck<check(..): Send> + Send + 'static,
{
    spawn(async move {
        let mut hc = hc;
        if !hc.check().await {
            log_health_check_failure().await;
        }
    });
}

async fn log_health_check_failure() {}

fn main() {}

// Fake tokio spawn

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

fn spawn<F>(future: F) -> JoinHandle
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    loop {}
}

struct JoinHandle;

impl Future for JoinHandle {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {}
    }
}
