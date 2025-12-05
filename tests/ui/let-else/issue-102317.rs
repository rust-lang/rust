// issue #102317
//@ build-pass
//@ compile-flags: -C opt-level=3 -Zvalidate-mir
//@ edition: 2021

struct SegmentJob;

impl Drop for SegmentJob {
    fn drop(&mut self) {}
}

pub async fn run() -> Result<(), ()> {
    let jobs = Vec::<SegmentJob>::new();
    let Some(_job) = jobs.into_iter().next() else {
        return Ok(())
    };

    Ok(())
}

fn main() {}
