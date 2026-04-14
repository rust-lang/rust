## Port wait semantics

Ports now expose two receive modes:

- `channel_recv`: blocking receive. If the queue is empty, the current task is parked on the port's read wait queue until data arrives or the write side closes.
- `port_try_recv`: non-blocking receive. If the queue is empty, it returns `EAGAIN`.

Queue and wake behavior:

- Ports remain single-consumer byte queues.
- Each successful send wakes one blocked reader.
- Each successful receive wakes one blocked writer.
- Wakeup is FIFO through the scheduler wait queue.

Close behavior:

- Closing the last write handle wakes blocked readers.
- A blocked reader that wakes to an empty queue with no writers left receives `EPIPE`.
- Closing the last read handle wakes blocked writers.
- A send on a port with no remaining readers returns `EPIPE`.
- When both sides are closed, the port is removed from the global registry.

This keeps raw ports as the waitable substrate for later `wait_many` style composition without embedding retry loops in userspace clients.
