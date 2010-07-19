#ifndef RUST_LOG_H
#define RUST_LOG_H

class rust_dom;
class rust_task;



class rust_log {

public:
    rust_log(rust_srv *srv, rust_dom *dom);
    virtual ~rust_log();

    enum ansi_color {
        WHITE,
        RED,
        LIGHTRED,
        GREEN,
        LIGHTGREEN,
        YELLOW,
        LIGHTYELLOW,
        BLUE,
        LIGHTBLUE,
        MAGENTA,
        LIGHTMAGENTA,
        TEAL,
        LIGHTTEAL
    };

    enum log_type {
        ERR = 0x1,
        MEM = 0x2,
        COMM = 0x4,
        TASK = 0x8,
        DOM = 0x10,
        ULOG = 0x20,
        TRACE = 0x40,
        DWARF = 0x80,
        CACHE = 0x100,
        UPCALL = 0x200,
        TIMER = 0x400,
        GC = 0x800,
        ALL = 0xffffffff
    };

    void indent();
    void outdent();
    void reset_indent(uint32_t indent);
    void trace_ln(char *prefix, char *message);
    void trace_ln(rust_task *task, uint32_t type_bits, char *message);
    void trace_ln(rust_task *task, ansi_color color, uint32_t type_bits, char *message);
    bool is_tracing(uint32_t type_bits);

private:
    rust_srv *_srv;
    rust_dom *_dom;
    uint32_t _type_bit_mask;
    bool _use_labels;
    bool _use_colors;
    uint32_t _indent;
    void trace_ln(rust_task *task, char *message);
};

#endif /* RUST_LOG_H */
