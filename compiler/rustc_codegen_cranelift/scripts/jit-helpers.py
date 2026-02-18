import gdb


def jitmap_raw():
    pid = gdb.selected_inferior().pid
    jitmap_file = open("/tmp/perf-%d.map" % (pid,), "r")
    jitmap = jitmap_file.read()
    jitmap_file.close()
    return jitmap


def jit_functions():
    jitmap = jitmap_raw()

    functions = []
    for line in jitmap.strip().split("\n"):
        [addr, size, name] = line.split(" ")
        functions.append((int(addr, 16), int(size, 16), name))

    return functions


class JitDecorator(gdb.FrameDecorator.FrameDecorator):
    def __init__(self, fobj, name):
        super(JitDecorator, self).__init__(fobj)
        self.name = name

    def function(self):
        return self.name


class JitFilter:
    """
    A backtrace filter which reads perf map files produced by cranelift-jit.
    """

    def __init__(self):
        self.name = "JitFilter"
        self.enabled = True
        self.priority = 0

        gdb.current_progspace().frame_filters[self.name] = self

    # FIXME add an actual unwinder or somehow register JITed .eh_frame with gdb to avoid relying on
    # gdb unwinder heuristics.
    def filter(self, frame_iter):
        for frame in frame_iter:
            frame_addr = frame.inferior_frame().pc()
            for addr, size, name in jit_functions():
                if frame_addr >= addr and frame_addr < addr + size:
                    yield JitDecorator(frame, name)
                    break
            else:
                yield frame


JitFilter()
