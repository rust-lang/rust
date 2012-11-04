#include <stdio.h>
#include <stdlib.h>
#include "linenoise.h"

#ifndef NO_COMPLETION
void completion(const char *buf, linenoiseCompletions *lc) {
    if (buf[0] == 'h') {
        linenoiseAddCompletion(lc,"hello");
        linenoiseAddCompletion(lc,"hello there");
    }
}
#endif

int main(void) {
    char *line;

#ifndef NO_COMPLETION
    linenoiseSetCompletionCallback(completion);
#endif
    linenoiseHistoryLoad("history.txt"); /* Load the history at startup */
    while((line = linenoise("hello> ")) != NULL) {
        if (line[0] != '\0') {
            printf("echo: '%s'\n", line);
            linenoiseHistoryAdd(line);
            linenoiseHistorySave("history.txt"); /* Save every new entry */
        }
        free(line);
    }
    return 0;
}
