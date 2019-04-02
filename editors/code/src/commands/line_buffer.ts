export class LineBuffer {
    private outBuffer: string = '';

    public processOutput(chunk: string, cb: (line: string) => void) {
        this.outBuffer += chunk;
        let eolIndex = this.outBuffer.indexOf('\n');
        while (eolIndex >= 0) {
            // line includes the EOL
            const line = this.outBuffer.slice(0, eolIndex + 1);
            cb(line);
            this.outBuffer = this.outBuffer.slice(eolIndex + 1);

            eolIndex = this.outBuffer.indexOf('\n');
        }
    }
}
