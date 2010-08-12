native "rust" mod rustrt {
  fn task_sleep(uint time_in_us);
}

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(uint time_in_us) {
  ret rustrt.task_sleep(time_in_us);
}