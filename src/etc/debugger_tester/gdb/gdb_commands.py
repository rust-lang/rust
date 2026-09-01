import gdb
import sys
import os

REPR_COMMAND_RUN = False
REPR_ERROR = False


class ReprCommand(gdb.Command):
    def __init__(self):
        super().__init__("repr", gdb.COMMAND_OBSCURE)

    def invoke(self, argument: str, from_tty: bool):
        from .check_gdb import check
        from ..common import Result

        print(f"(gdb) repr {argument}")

        global REPR_COMMAND_RUN
        REPR_COMMAND_RUN = True
        try:
            if check(argument) == Result.Mismatch:
                global REPR_ERROR
                REPR_ERROR = True
        except Exception as e:
            import sys
            import traceback

            traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)
            gdb.execute("exit 1")


ReprCommand()


class ReprFinalize(gdb.Command):
    def __init__(self):
        super().__init__("repr_finalize", gdb.COMMAND_OBSCURE)

    def invoke(self, argument: str, from_tty: bool):
        if not REPR_COMMAND_RUN:
            return

        from ..common import (
            BLESS,
            INPUT_DATA,
            BlessMetadata,
            tested_all_types,
            tested_all_variables,
        )

        gdb_version = gdb.execute("show version", to_string=True).splitlines()[0]

        if (
            REPR_ERROR
            and (is_ci := os.environ.get("CI")) is not None
            and is_ci == "true"
        ):
            from lldb_providers import FEATURE_FLAGS

            path = os.path.relpath(os.environ["DEBUGGER_TESTER_INPUT_DATA_PATH"])

            print(f"[repr] If you do not have access to this target, you can manually update \
the test data by overwriting the data in {path} with the following:")

            INPUT_DATA.print_json(
                BlessMetadata(sys.version, gdb_version, str(FEATURE_FLAGS))
            )

        if not tested_all_variables() or not tested_all_types():
            gdb.execute("exit 1")

        if BLESS and not REPR_ERROR:
            metadata = BlessMetadata(sys.version, gdb_version)
            INPUT_DATA.save_blessing(metadata)


ReprFinalize()
