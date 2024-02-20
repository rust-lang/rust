@REM A very basic test output postprocessor. Used in `test_output_postprocessing()`.

@echo off

if [%TEST_POSTPROCESSOR_OUTPUT_FILE%] == [] (
    echo Required environment variable TEST_POSTPROCESSOR_OUTPUT_FILE is not set.
    cmd /C exit /B 1
)

@REM Forward script's input into file.
find /v "" > %TEST_POSTPROCESSOR_OUTPUT_FILE%

@REM Log every command line argument into the same file.
:start
    if [%1] == [] goto done
    echo %~1>> %TEST_POSTPROCESSOR_OUTPUT_FILE%
    shift
    goto start
:done
