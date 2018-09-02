use std::thread;
use drop_bomb::DropBomb;
use Result;

pub struct ThreadWatcher {
    name: &'static str,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

impl ThreadWatcher {
    pub fn spawn(name: &'static str, f: impl FnOnce() + Send + 'static) -> ThreadWatcher {
        let thread = thread::spawn(f);
        ThreadWatcher {
            name,
            thread,
            bomb: DropBomb::new(format!("ThreadWatcher {} was not stopped", name)),
        }
    }

    pub fn stop(mut self) -> Result<()> {
        info!("waiting for {} to finish ...", self.name);
        let name = self.name;
        self.bomb.defuse();
        let res = self.thread.join()
            .map_err(|_| format_err!("ThreadWatcher {} died", name));
        match &res {
            Ok(()) => info!("... {} terminated with ok", name),
            Err(_) => error!("... {} terminated with err", name)
        }
        res
    }
}
