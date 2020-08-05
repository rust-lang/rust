//! Module providing a helper structure to capture output in subprocesses.

use std::{
    io,
    io::prelude::Write,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub struct Sink(Arc<Mutex<Vec<u8>>>);

impl Sink {
    pub fn new_boxed(data: &Arc<Mutex<Vec<u8>>>) -> Box<Self> {
        Box::new(Self(data.clone()))
    }
}

impl io::LocalOutput for Sink {
    fn clone_box(&self) -> Box<dyn io::LocalOutput> {
        Box::new(self.clone())
    }
}

impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Write::write(&mut *self.0.lock().unwrap(), data)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
