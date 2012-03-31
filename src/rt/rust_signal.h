#ifndef RUST_SIGNAL_H
#define RUST_SIGNAL_H

// Just an abstrict class that reperesents something that can be signalled
class rust_signal {
public:
    virtual void signal() = 0;
};

#endif /* RUST_SIGNAL_H */
