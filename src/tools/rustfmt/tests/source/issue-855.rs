fn main() {
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
            }
        }
    }
}

fn main2() {
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDownXXXXXXXXXXXXX { keycode: Some(Keycode::Escape), .. } => break 'running,
            }
        }
    }
}
