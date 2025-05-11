//@ known-bug: #127033
//@ edition: 2021

pub trait RaftLogStorage {
    fn save_vote(vote: ()) -> impl std::future::Future + Send;
}

struct X;
impl RaftLogStorage for X {
    fn save_vote(vote: ()) -> impl std::future::Future {
        loop {}
        async {
            vote
        }
    }
}

fn main() {}
