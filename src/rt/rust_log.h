#ifndef RUST_LOG_H_
#define RUST_LOG_H_

class rust_dom;

class rust_log {
    rust_srv *_srv;
    rust_dom *_dom;
    uint32_t _type_bit_mask;
    bool _use_colors;
    uint32_t _indent;
    void trace_ln(char *message);
public:
    rust_log(rust_srv *srv, rust_dom *dom);
    virtual ~rust_log();

    enum ansi_color {
        BLACK,
        GRAY,
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
    void trace_ln(uint32_t type_bits, char *message);
    void trace_ln(ansi_color color, uint32_t type_bits, char *message);
    bool is_tracing(uint32_t type_bits);
    static ansi_color get_type_color(log_type type);
};

#endif /* RUST_LOG_H_ */
