#!/bin/bash
#
# A very basic test output postprocessor. Used in `test_output_postprocessing()`.

if [ -z "$TEST_POSTPROCESSOR_OUTPUT_FILE" ]
then
  echo "Required environment variable TEST_POSTPROCESSOR_OUTPUT_FILE is not set."
  exit 1
fi

# Forward script's input into file.
cat /dev/stdin > "$TEST_POSTPROCESSOR_OUTPUT_FILE"

# Log every command line argument into the same file.
for i in "$@"
do
  echo "$i" >> "$TEST_POSTPROCESSOR_OUTPUT_FILE"
done
